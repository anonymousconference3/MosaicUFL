from .utils.lp import *
import math, numpy

from random import Random
import random
from collections import OrderedDict
import logging
import numpy as np2

def create_training_selector(args):
    return _training_selector(args)

def create_testing_selector(data_distribution=None, client_info=None, model_size=None):
    return _testing_selector(data_distribution, client_info, model_size)

class _testing_selector:
    """Oort's testing selector

    We provide two kinds of selector:
    select_by_deviation: testing participant selection that preserves data representativeness.
    select_by_category: testing participant selection that enforce developer's requirement on
        distribution of the testing set. Note that this selector is avaliable only if the client
        info is provided.

    Attributes:
        client_info: Optional; A dictionary that stores client id to client profile(system speech and
            network bandwidth) mapping. For example, {1: [153.0, 2209.61]} indicates that client 1
            needs 153ms to run a single sample inference and their network bandwidth is 2209 Kbps.
        model_size: Optional; the size of the model(i.e., the data transfer size) in kb
        data_distribution: Optional; individual data characteristics(distribution).
    """
    def __init__(self, data_distribution=None, client_info=None, model_size=None):
        """Inits testing selector."""
        self.client_info = client_info
        self.model_size = model_size
        self.data_distribution = data_distribution
        if self.client_info:
            self.client_idx_list = list(range(len(client_info)))


    def update_client_info(self, client_ids, client_profile):
        """Update clients' profile(system speed and network bandwidth)

        Since the clients' info is dynamic, developers can use this function
        to update clients' profile. If the client id does not exist, Oort will
        create a new entry for this client.

        Args:
            client_ids: A list of client ids whose profile needs to be updated
            client_info: Updated information about client profile, formatted as
                a list of pairs(speed, bw)

        Raises:
            Raises an error if len(client_ids) != len(client_info)
        """
        return 0

    def _hoeffding_bound(self, dev_tolerance, capacity_range, total_num_clients, confidence=0.8):
        """Use hoeffding bound to cap the deviation from E[X]

        Args:
            dev_tolerance: maximum deviation from the empirical (E[X])
            capacity_range: the global max-min range of number of samples across all clients
            total_num_clients: total number of feasible clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence

        Returns:
            The estimated number of participant needed to satisfy developer's requirement
        """

        factor = (1.0 - 2*total_num_clients/math.log(1-math.pow(confidence, 1)) \
                                    * (dev_tolerance/float(capacity_range)) ** 2)
        n = (total_num_clients+1.0)/factor

        return n

    def select_by_deviation(self, dev_target, range_of_capacity, total_num_clients,
            confidence=0.8, overcommit=1.1):
        """Testing selector that preserves data representativeness.

        Given the developer-specified tolerance `dev_target`, Oort can estimate the number
        of participants needed such that the deviation from the representative categorical
        distribution is bounded.

        Args:
            dev_target: developer-specified tolerance
            range_of_capacity: the global max-min range of number of samples across all clients
            confidence: Optional; Pr[|X - E[X]| < dev_tolerance] > confidence
            overcommit: Optional; to handle stragglers

        Returns:
            A list of selected participants
        """
        num_of_selected = self._hoeffding_bound(dev_target, range_of_capacity, total_num_clients, confidence=0.8)
        #selected_client_ids = numpy.random.choice(self.client_idx_list, replacement=False, size=num_of_selected*overcommit)
        return num_of_selected

    def select_by_category(self, request_list, max_num_clients=None, greedy_heuristic=True):
        """Testing selection based on requested number of samples per category.

        When individual data characteristics(distribution) is provided, Oort can
        enforce client's request on the number of samples per category.

        Args:
            request_list: a list that specifies the desired number of samples per category.
                i.e., [num_requested_samples_class_x for class_x in request_list].
            max_num_clients: Optional; the maximum number of participants .
            greedy_heuristic: Optional; whether to use Oort-based solver. Otherwise, Mix-Integer Linear Programming
        Returns:
            A list of selected participants ids.

        Raises:
            Raises an error if 1) no client information is provided or 2) the requirement
            cannot be satisfied(e.g., max_num_clients too small).
        """
        # TODO: Add error handling
        client_sample_matrix, test_duration, lp_duration = run_select_by_category(request_list, self.data_distribution,
            self.client_info, max_num_clients, self.model_size, greedy_heuristic)
        return client_sample_matrix, test_duration, lp_duration


class _training_selector(object):
    """Oort's training selector
    """
    def __init__(self, args, sample_seed=233):

        self.totalArms = OrderedDict()
        self.training_round = 0

        self.exploration = args.exploration_factor
        self.decay_factor = args.exploration_decay
        self.exploration_min = args.exploration_min
        self.alpha = args.exploration_alpha

        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        self.args = args
        self.round_threshold = args.round_threshold
        self.round_prefer_duration = float('inf')
        self.last_util_record = 0

        self.sample_window = self.args.sample_window
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        self.blacklist = None

        np2.random.seed(sample_seed)

    def register_client(self, clientId, feedbacks):
        # Initiate the score for arms. [score, time_stamp, # of trials, size of client, auxi, duration]
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {}
            self.totalArms[clientId]['reward'] = feedbacks['reward']
            self.totalArms[clientId]['duration'] = feedbacks['duration']
            self.totalArms[clientId]['time_stamp'] = self.training_round
            self.totalArms[clientId]['count'] = 0
            self.totalArms[clientId]['status'] = True

            self.unexplored.add(clientId)

    def calculateSumUtil(self, clientList):
        cnt, cntUtil = 1e-4, 0

        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']

        return cntUtil/cnt

    def pacer(self):
        # summarize utility in last epoch
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)

        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)

        self.successfulClients = set()

        if self.training_round >= 2 * self.args.pacer_step and self.training_round % self.args.pacer_step == 0:

            utilLastPacerRounds = sum(self.exploitUtilHistory[-2*self.args.pacer_step:-self.args.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.args.pacer_step:])

            # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            # change sharply -> we decrease the pacer step
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.args.pacer_delta, self.round_threshold - self.args.pacer_delta)
                self.last_util_record = self.training_round - self.args.pacer_step
                logging.debug("Training selector: Pacer changes at {} to {}".format(self.training_round, self.round_threshold))

            logging.debug("Training selector: utilLastPacerRounds {}, utilCurrentPacerRounds {} in round {}"
                .format(utilLastPacerRounds, utilCurrentPacerRounds, self.training_round))

        logging.info("Training selector: Pacer {}: lastExploitationUtil {}, lastExplorationUtil {}, last_util_record {}".
                        format(self.training_round, lastExploitationUtil, lastExplorationUtil, self.last_util_record))

    def update_client_util(self, clientId, feedbacks):
        '''
        @ feedbacks['reward']: statistical utility
        @ feedbacks['duration']: system utility
        @ feedbacks['count']: times of involved
        '''
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = feedbacks['status']
        logging.info(self.totalArms[clientId]['reward'])
        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)


    def get_blacklist(self):
        blacklist = []

        if self.args.blacklist_rounds != -1:
            sorted_client_ids = sorted(list(self.totalArms), reverse=True,
                                        key=lambda k:self.totalArms[k]['count'])

            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.args.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break

            # we need to back up if we have blacklisted all clients
            predefined_max_len = self.args.blacklist_max_len * len(self.totalArms)

            if len(blacklist) > predefined_max_len:
                logging.warning("Training Selector: exceeds the blacklist threshold")
                blacklist = blacklist[:predefined_max_len]

        return set(blacklist)

    def select_participant(self, num_of_clients, feasible_clients=None):
        '''
        @ num_of_clients: # of clients selected
        '''
        #feasible_clients is not None
        viable_clients = feasible_clients if feasible_clients is not None else set([x for x in self.totalArms.keys() if self.totalArms[x]['status']])
        return self.getTopK(num_of_clients, self.training_round+1, viable_clients)

    def update_duration(self, clientId, duration):
        if clientId in self.totalArms:
            self.totalArms[clientId]['duration'] = duration

    def checkClients(self,feasible_clients):
        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if int(x) in feasible_clients]
        for clientId in orderedKeys:
            logging.info("check reward {}".format(self.totalArms[clientId]['reward']))
            if self.totalArms[clientId]['reward'] <= 0:
                return False
        return True

    def getTopK(self, numOfSamples, cur_time, feasible_clients):
        self.training_round = cur_time
        self.blacklist = self.get_blacklist()

        self.pacer()

        # normalize the score of all arms: Avg + Confidence
        scores = {}
        numOfExploited = 0
        exploreLen = 0
        
        client_list = list(self.totalArms.keys())
        orderedKeys = [x for x in client_list if int(x) in feasible_clients and int(x) not in self.blacklist]
        
        #orderedKeys: all clients
        if self.round_threshold < 100.:
            #here
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            #opt_time
            # duration_key_list = [(self.totalArms[key]['duration'], key) for key in client_list]

            # sortedDuration = sorted(duration_key_list)
            # logging.info(f"opt-speed: {sortedDuration}")
            # client_output = []
            # i = 0
            # while i < numOfSamples:
            #     client_output.append(sortedDuration[i][1])
            #     i+=1
            # return client_output


            #opt-stat
            # reward_key_list = [(self.totalArms[key]['reward'], key) for key in client_list]
            # sortedreward = sorted(reward_key_list, reverse=True)
            # logging.info(f"opt-stat: {sortedreward}")
            # client_output = []
            # i = 0
            # while i < numOfSamples:
            #     if sortedreward[i][0] != sortedreward[99][0]:
            #         client_output.append(sortedreward[i][1])
            #     i+=1
            # if len(client_output) != numOfSamples:
            #     rest = numOfSamples - len(client_output)
            #     random_list = random.sample(range(len(client_output), 100), rest)
            #     for each in random_list:
            #         client_output.append(sortedreward[each][1])
            # return client_output
            self.round_prefer_duration = sortedDuration[min(int(len(sortedDuration) * self.round_threshold/100.), len(sortedDuration)-1)]
        else:
            logging.info("bad")
            self.round_prefer_duration = float('inf')
       
        moving_reward, staleness, allloss = [], [], {}

        for clientId in orderedKeys:
            logging.info("reward {} {} {}".format(clientId, self.totalArms[clientId]['reward'],self.totalArms[clientId]['duration']))
            if self.totalArms[clientId]['reward'] > 0:
                creward = self.totalArms[clientId]['reward']
                moving_reward.append(creward)
                staleness.append(cur_time - self.totalArms[clientId]['time_stamp'])
                


        max_reward, min_reward, range_reward, avg_reward, clip_value = self.get_norm(moving_reward, self.args.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = self.get_norm(staleness, thres=1)

        for key in orderedKeys:
            # we have played this arm before
            if self.totalArms[key]['count'] > 0:
                
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1

                sc = (creward - min_reward)/float(range_reward) \
                     + math.sqrt(0.1*math.log(cur_time)/self.totalArms[key]['time_stamp']) # temporal uncertainty
                
                #sc = (creward - min_reward)/float(range_reward) \
                #    + self.alpha*((cur_time-self.totalArms[key]['time_stamp']) - min_staleness)/float(range_staleness)

                clientDuration = self.totalArms[key]['duration']
                logging.info(f"checks lientDuration{clientDuration}")
                if clientDuration > self.round_prefer_duration:
                    #time punish here
                    a = sc
                    sc *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)
                    logging.info(f"checks {creward, numOfExploited,min_reward, a,((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty),sc}")

                if self.totalArms[key]['time_stamp']==cur_time:
                    allloss[key] = sc

                scores[key] = sc


        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration*self.decay_factor, self.exploration_min)
        exploitLen = min(int(numOfSamples*(1.0 - self.exploration)), len(clientLakes))

        # take the top-k, and then sample by probability, take 95% of the cut-off loss
        sortedClientUtil = sorted(scores, key=scores.get, reverse=True)

        # take cut-off utility
        cut_off_util = scores[sortedClientUtil[exploitLen]] * self.args.cut_off_util
        logging.info(f"sortedClientUtil: {sortedClientUtil}")
        logging.info(f"cut_off_util: {cut_off_util}")
        pickedClients = []
        for clientId in sortedClientUtil:
            if scores[clientId] < cut_off_util:
                break
            pickedClients.append(clientId)
        logging.info(f"pickedClients: {pickedClients}")
        augment_factor = len(pickedClients)

        totalSc = max(1e-4, float(sum([scores[key] for key in pickedClients])))
        pickedClients = list(np2.random.choice(pickedClients, exploitLen, p=[scores[key]/totalSc for key in pickedClients], replace=False))
        self.exploitClients = pickedClients

        # exploration
        if len(self.unexplored) > 0:
            _unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients]

            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']

                if clientDuration > self.round_prefer_duration:
                    init_reward[cl] *= ((float(self.round_prefer_duration)/max(1e-4, clientDuration)) ** self.args.round_penalty)

            # prioritize w/ some rewards (i.e., size)
            exploreLen = min(len(_unexplored), numOfSamples - len(pickedClients))
            pickedUnexploredClients = sorted(init_reward, key=init_reward.get, reverse=True)[:min(int(self.sample_window*exploreLen), len(init_reward))]

            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))

            pickedUnexplored = list(np2.random.choice(pickedUnexploredClients, exploreLen,
                            p=[init_reward[key]/max(1e-4, unexploredSc) for key in pickedUnexploredClients], replace=False))

            self.exploreClients = pickedUnexplored
            pickedClients = pickedClients + pickedUnexplored
        else:
            # no clients left for exploration
            self.exploration_min = 0.
            self.exploration = 0.

        while len(pickedClients) < numOfSamples:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)

        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            clientId = pickedClients[i]
            _score = (self.totalArms[clientId]['reward'] - min_reward)/range_reward
            _staleness = self.alpha*((cur_time-self.totalArms[clientId]['time_stamp']) - min_staleness)/float(range_staleness) #math.sqrt(0.1*math.log(cur_time)/max(1e-4, self.totalArms[clientId]['time_stamp']))
            top_k_score.append((self.totalArms[clientId], [_score, _staleness]))

        logging.info("At round {}, UCB exploited {}, augment_factor {}, exploreLen {}, un-explored {}, exploration {}, round_threshold {}, sampled score is {}"
            .format(cur_time, numOfExploited, augment_factor/max(1e-4, exploitLen), exploreLen, len(self.unexplored), self.exploration, self.round_threshold, top_k_score))
        # logging.info("At time {}, all rewards are {}".format(cur_time, allloss))

        return pickedClients

    def get_median_reward(self):
        feasible_rewards = [self.totalArms[x]['reward'] for x in list(self.totalArms.keys()) if int(x) not in self.blacklist]

        # we report mean instead of median
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards)/float(len(feasible_rewards))

        return 0

    def get_client_reward(self, armId):
        return self.totalArms[armId]

    def getAllMetrics(self):
        return self.totalArms

    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        aList.sort()
        clip_value = aList[min(int(len(aList)*clip_bound), len(aList)-1)]

        _max = max(aList)
        _min = min(aList)*0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList)/max(1e-4, float(len(aList)))

        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
