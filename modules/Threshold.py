
class AccountThreshold:
    def __init__(self, name, attributes, total_friends, age_of_account, friendship_duration, mutual_friends):
        self.name = name
        self.attributes = attributes
        self.total_friends = total_friends
        self.age_of_account = age_of_account
        self.friendship_duration = friendship_duration
        self.mutual_friends = mutual_friends


class PageThreshold:
    def __init__(self,page_name, age_of_page, attributes, followers, num_of_likes, mutual_friends):
        self.page_name = page_name
        self.age_of_page = age_of_page
        self.attributes = attributes
        self.followers = followers
        self.num_of_likes = num_of_likes
        self.mutual_friends = mutual_friends


class GroupThreshold:
    def __init__(self, group_name, attributes, age_of_group, friends, mutual_friends):
        self.group_name = group_name
        self.attributes = attributes
        self.age_of_group = age_of_group
        self.friends = friends
        self.mutual_friends = mutual_friends




