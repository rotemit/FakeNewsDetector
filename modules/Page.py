from .Threshold import PageThreshold


class Page:
    def __init__(self, page_name, age_of_page, attributes, followers, num_of_likes, mutual_friends=0):
        self.page_name = page_name
        self.age_of_page = age_of_page
        self.attributes = attributes
        self.followers = followers
        self.num_of_likes = num_of_likes
        self.mutual_friends = mutual_friends

    def __str__(self):
        return "Name: " + self.page_name + "\nAge: " + str(self.age_of_page) + "\nFollowers: " + str(self.followers) + \
               "\nNumber of likes:  " + str(self.num_of_likes) + "\nMutual Friends: " + str(self.mutual_friends) + \
                "\nPosts: " + str(self.posts)

    def set_posts(self, posts_arr):
        self.posts = posts_arr

    def set_trust_value(self, threshold: PageThreshold):

        total_friends_param = 1 if self.followers > threshold.followers \
            else self.followers / threshold.followers

        age_of_account_param = 1 if self.age_of_page > threshold.age_of_page \
            else self.age_of_page / threshold.age_of_page

        account_trust_value = (total_friends_param + age_of_account_param) / 2

        mutual_friends_param = 1 if self.mutual_friends > threshold.mutual_friends \
            else self.mutual_friends / threshold.mutual_friends

        connection_trust_value = mutual_friends_param / 2
        self.page_trust_value = (account_trust_value + connection_trust_value) / 2
