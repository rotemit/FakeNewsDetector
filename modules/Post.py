class Post:
    def __init__(self, writer, content, comments, account):
        self.writer = writer
        self.content = content
        self.comments = comments
        self.account = account


    def __str__(self):
        return "Writer: " + self.writer + "\nContent: " + self.content + "\nComments: " + str(self.comments) + \
                "\nWriter_Accounr: " + str(self.account)
