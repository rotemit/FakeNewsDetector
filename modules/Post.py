class Post:
    def __init__(self, writer, content, account):
        self.writer = writer
        self.content = content
        self.account = account


    def __str__(self):
        return "Writer: " + self.writer + "\nContent: " + self.content + "\nWriter_Accounr: " + str(self.account)
