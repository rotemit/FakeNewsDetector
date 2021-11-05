from modules.AnalysisResult import AnalysisResult

# the output of the analyzers - manager of all the analysis
# gathers all outputs of all aalysis
class ScanResult:
    user_name: str
    sentimentAnalyzer_result: AnalysisResult
    machineLearning_result: AnalysisResult
    utv_result: AnalysisResult


    def __init__(self, user_name, sentimentAnalyzer_result: AnalysisResult, machineLearning_result: AnalysisResult, utv_result: AnalysisResult):
        self.user_name = user_name
        self.sentimentAnalyzer_result = sentimentAnalyzer_result
        self.machineLearning_result = machineLearning_result
        self.utv_result = utv_result

    def __str__(self):
        return "Name: " + self.user_name + "\nSentiment Analyzer Result:\n" + str(self.sentimentAnalyzer_result) + \
               "\nMachine Learning Result:\n" + str(self.machineLearning_result)+ "\nTrust Value Result:\n" + str(self.utv_result)


    def __dict__(self):
        return {"user_name":self.user_name, "sentimentAnalyzer_result":vars(self.sentimentAnalyzer_result), "machineLearning_result":vars(self.machineLearning_result), "utv_result":vars(self.utv_result)}