from time import perf_counter

class Time_measure():

    @staticmethod
    def starttimer(text, seperator='!--!'):
        print(seperator + ' ' + text + ' ' + seperator)
        return perf_counter()

    @staticmethod
    def stoptimer(startTimer, add_text:str = ''):
        stopTimer = perf_counter()
        print(add_text +' takes ', stopTimer - startTimer, ' time in seconds')
