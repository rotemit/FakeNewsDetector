class Page:
    def __init__(self, name, age_of_page, followers, num_of_likes, mutual_friends=0):
        self.name = name
        self.age = age_of_page
        self.followers = followers
        self.num_of_likes = num_of_likes
        self.mutual_friends = mutual_friends
        self.posts = []

    def __str__(self):
        return "Name: " + self.name + "\nAge: " + str(self.age) + "\nFollowers: " + str(self.followers) + \
               "\nNumber of likes:  " + str(self.num_of_likes) + "\nMutual Friends: " + str(self.mutual_friends) + \
                "\nPosts: " + str(self.posts)

    def set_posts(self, posts_arr):
        self.posts = posts_arr
