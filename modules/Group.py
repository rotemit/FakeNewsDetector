class Group:
    def __init__(self, name, age_of_group, friends, mutual_friends=0):
        self.name = name
        self.age = age_of_group
        self.friends = friends
        self.mutual_friends = mutual_friends
        self.posts = []

    def set_posts(self, posts_arr):
        self.posts = posts_arr

    def __str__(self):
        return  "Name: " + self.name + "\nGroup Age: " + str(self.age) + \
                "\nNumber of Friends: " + str(self.friends) + "\nNumber of Mutual Friends: " + str(self.mutual_friends) + \
                "\nPosts: " + str(self.posts)