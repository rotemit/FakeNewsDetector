import json
from collections import namedtuple

class Account:
    def __init__(self, name, attributes, total_friends, age_of_account, friendship_duration=0, mutual_friends=0):
        self.name = name
        self.attributes = attributes
        self.total_friends = total_friends
        self.age = age_of_account
        self.friendship_duration = friendship_duration
        self.mutual_friends = mutual_friends
        self.posts = []

    def __str__(self):
        return "Name: " + str(self.name) + "\n" \
               + "attributes: " + str(self.attributes) + "\nFriendship Duration: " + str(self.friendship_duration) \
               + "\nNumber of Mutual Friends: " + str(self.mutual_friends) + "\nTotal Friends: " + \
               str(self.total_friends) + "\nAge Of Account:  " + str(self.age) + \
                "\nposts: " + str(self.posts)

    def set_posts(self, posts_arr):
        self.posts = posts_arr


class account_encoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__
