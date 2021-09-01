# The output of each analysis
class AnalysisResult:
    percent: str
    text: str
    numeric: float

    def __init__(self, percetResult, textResult, numericResult):
        self.percent = percetResult
        self.text = textResult
        self.numeric = numericResult


    def __str__(self):
        return "Percent: " + self.percent + "\nText: " + self.text + "\nNumeric: " + str(self.numeric)