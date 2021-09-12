from .Threshold import AccountThreshold
import json
from collections import namedtuple


class Account:
    def __init__(self, name, attributes, total_friends, age_of_account, friendship_duration=0, mutual_friends=0):
        self.name = name
        self.attributes = attributes
        self.total_friends = total_friends
        self.age_of_account = age_of_account
        self.friendship_duration = friendship_duration
        self.mutual_friends = mutual_friends

    def __str__(self):
        return "Name: " + str(self.name) + "\n" \
               + "attributes: " + str(self.attributes)+ "\nFriendship Duration: " + str(self.friendship_duration) \
               + "\nNumber of Mutual Friends: " + str(self.mutual_friends) + "\nTotal Friends: " + \
                str(self.total_friends) + "\nAge Of Account:  " + str(self.age_of_account) + \
                "\nposts: " + str(self.posts)

    def set_posts(self, posts_arr):
        self.posts = posts_arr

    def calc_resemblance_attributes(self, ego_node_attributes):
        resemblance_attributes_counter = 0
        for field in ego_node_attributes:
            if field in self.attributes.keys():
                for info in ego_node_attributes[field]:
                    if self.attributes[field].find(info) != -1:
                        resemblance_attributes_counter += 1
                        break
        return resemblance_attributes_counter


    def set_trust_value(self, threshold: AccountThreshold):
        # Setting UserTrustValue

        total_friends_param = 1 if self.total_friends > threshold.total_friends \
            else self.total_friends / threshold.total_friends

        age_of_account_param = 1 if self.age_of_account > threshold.age_of_account \
            else self.age_of_account / threshold.age_of_account

        account_trust_value = (total_friends_param + age_of_account_param) / 2

        friendship_duration_param = 1 if self.friendship_duration > threshold.friendship_duration \
            else self.friendship_duration / threshold.friendship_duration

        mutual_friends_param = 1 if self.mutual_friends > threshold.mutual_friends \
            else self.mutual_friends / threshold.mutual_friends

        resemblance_attributes_param = self.calc_resemblance_attributes(threshold.attributes) \
                                    / len(threshold.attributes)


        default_param_addition = mutual_friends_param + resemblance_attributes_param

        connection_trust_value = (default_param_addition + friendship_duration_param) / 3 \
            if friendship_duration_param != 0 else default_param_addition / 2
        self.account_trust_value = (account_trust_value + connection_trust_value) / 2

    @classmethod
    def from_json(cls, json_string):
        json_dict = json.loads(json_string)
        return cls(**json_dict)


class account_encoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def account_decoder(account_dict):
    return namedtuple('X', account_dict.keys())()
