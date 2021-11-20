class Group:
    def __init__(self, name, age_of_group, friends, isPrivate, isVisible, mutual_friends):
        self.name = name
        self.age = age_of_group
        self.friends = friends
        self.isPrivate = isPrivate
        self.isVisible = isVisible
        self.mutual_friends = mutual_friends
        self.posts = []

    def set_posts(self, posts_arr):
        self.posts = posts_arr

    def __str__(self):
        return  "Name: " + self.name + "\nGroup Age: " + str(self.age) + "\nPrivate: " + str(bool(self.isPrivate)) + \
                "\nVisible: " + str(bool(self.isVisible)) + "\nNumber of Friends: " + str(self.friends) + \
                "\nNumber of Mutual Friends: " + str(self.mutual_friends) + "\nPosts: " + str(self.posts)