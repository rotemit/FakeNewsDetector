class Group:
    def __init__(self, name, attributes, age_of_group, friends, mutual_friends=0):
        self.name = name
        self.attributes = attributes
        self.age_of_group = age_of_group
        self.friends = friends
        self.mutual_friends = mutual_friends
        self.posts = []

    def set_posts(self, posts_arr):
        self.posts = posts_arr

    def __str__(self):
        return  "Name: " + self.name + "\nattributes: " + str(self.attributes) + "\nGroup Age: " + str(self.age_of_group) + \
                "\nNumber of Friends: " + str(self.friends) + "\nNumber of Mutual Friends: " + str(self.mutual_friends) + \
                "\nPosts: " + str(self.posts)