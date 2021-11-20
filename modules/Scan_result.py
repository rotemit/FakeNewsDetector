# the output of the analyzers - manager of all the analysis
# gathers all outputs of all analysis
class ScanResult:
    name: str
    sentimentAnalyzer_result: float
    machineLearning_result: float
    utv_result: float

    def __init__(self, user_name, sentimentAnalyzer_result, machineLearning_result, utv_result):
        self.name = user_name
        self.sentimentAnalyzer_result = sentimentAnalyzer_result
        self.machineLearning_result = machineLearning_result
        self.utv_result = utv_result

    def __str__(self):
        return "Name: " + self.name + "\nSentiment Analyzer Result:\n" + str(self.sentimentAnalyzer_result) + \
               "\nMachine Learning Result:\n" + str(self.machineLearning_result) + "\nTrust Value Result:\n" + str(self.utv_result)