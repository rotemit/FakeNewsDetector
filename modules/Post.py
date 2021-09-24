class Post:
    def __init__(self, writer, content, comments):
        self.writer = writer
        self.content = content
        self.comments = comments


    def __str__(self):
        return "Writer: " + self.writer + "\nContent: " + self.content + "\nComments: " + str(self.comments)
