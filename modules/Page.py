class Page:
    def __init__(self, name, age_of_page, attributes, followers, num_of_likes, mutual_friends=0):
        self.name = name
        self.age_of_page = age_of_page
        self.attributes = attributes
        self.followers = followers
        self.num_of_likes = num_of_likes
        self.mutual_friends = mutual_friends
        self.posts = []

    def __str__(self):
        return "Name: " + self.name + "\nAge: " + str(self.age_of_page) + "\nFollowers: " + str(self.followers) + \
               "\nNumber of likes:  " + str(self.num_of_likes) + "\nMutual Friends: " + str(self.mutual_friends) + \
                "\nPosts: " + str(self.posts)

    def set_posts(self, posts_arr):
        self.posts = posts_arr
