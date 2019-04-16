# Copyright 2019 AIST
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
from typing import Any, Callable, Tuple

import numpy as np
import zmq


def str2Bool(recvStr):
    if recvStr.find("true") != -1:
        return True
    else:
        return False


class PyLabClient(object):
    def __init__(self, binder: Any, paramHeader: np.ndarray, resultHeader: np.ndarray) -> None:
        #zeroMQ socket client
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(binder)

        #headers (arrays storing name of parameters/results)
        self.paramHeader = paramHeader
        self.resultHeader = resultHeader
        self.paramHeaderReceived = False
        self.resultHeaderReceived = False
        #Last experiment (parameters and results) sent to learner
        self.lastExpParam = np.array([0], dtype = float)
        self.lastExpResult = np.array([0], dtype = float)
        self.lastExpReady = False
        #Next parameters from learner
        self.nextExpParam = np.array([0], dtype = float)
        self.nextExpReceived = False
        #counter
        self.counter = 0
        
    def receiveParamHeaderAsStr(self, paramHeaderStr):
        self.paramHeader = paramHeaderStr.split("\t")
        self.paramHeaderReceived = True
        return self.sendParamHeaderAsStr()

    def sendParamHeaderAsStr(self):
        headerStr = ""
        for hstr in self.paramHeader:
            headerStr += hstr
            headerStr += "\t"
        return headerStr.rstrip("\t")
    
    def receiveResultHeaderAsStr(self, resultHeaderStr):
        self.resultHeader = resultHeaderStr.split("\t")
        self.resultHeaderReceived = True
        return self.sendResultHeaderAsStr()

    def sendResultHeaderAsStr(self):
        headerStr = ""
        for hstr in self.resultHeader:
            headerStr += hstr
            headerStr += "\t"            
        return headerStr.rstrip("\t")
    
    def isDammyExperimentReady(self):
        return self.nextExpReceived and not self.lastExpReady

    #Dammy experimenter conduct new experiment 
    def doDammyExperiment(self, experiment: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> None:
        numParam = self.paramHeader.size
        numResult = self.resultHeader.size
        self.lastExpParam, self.lastExpResult = experiment(self.nextExpParam)
        self.lastExpReady = True
        self.nextExpReceived = False
        print("%d th experiment was done." % self.counter)
        self.counter += 1

    #get initial (dammy) parameters from experimentor
    def getInitialParameters(self):
        numParam = self.paramHeader.size
        numResult = self.resultHeader.size
        self.lastExpParam = np.full(numParam, 0, dtype = np.float)
        self.lastExpResult = np.full(numResult, 0, dtype = np.float)
        self.lastExpReady = True
        self.nextExpReceived = False

    def receiveNextExpAsStr(self, recvStr):
        self.nextExpReceived = True
        self.nextExpParam = self.str2NParray(recvStr)
        print(self.nextExpParam)
    
    def sendLastExpAsStr(self):
        sendStr = self.npArray2Str(self.lastExpParam)
        sendStr += "\", \""
        sendStr += self.npArray2Str(self.lastExpResult)
        self.lastExpReady = False
        return sendStr
    
    @staticmethod
    def npArray2Str(npArray):
        npAsStr = ""
        for num in npArray:
            npAsStr += str(num)
            npAsStr += "\t"
        return npAsStr.rstrip("\t")
    
    #slice the camma separated array str into nparray
    @staticmethod
    def str2NParray(arrayStr):
        #print arrayStr.split(",")
        list = []
        for valStr in arrayStr.lstrip("\"").rstrip("\"").split("\\t"):
            try:
                valNum = float(valStr)
            except:
                valNum = float("NaN")
            list.append(valNum)
        npArray = np.array(list)
        return npArray

    @classmethod
    def run(cls, binder: Any, paramHeader: np.ndarray, resultHeader: np.ndarray, experiment: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]], WAITTIME: float = 0.2) -> None:

        # Socket to talk to PyLabZMQ server
        print("connecting to PyLabZMQ server")
        self = cls(binder, paramHeader, resultHeader)

        try:
            #Tell sequencer is stopped
            self.socket.send_string("sequencerStopped()")
            print(self.socket.recv_string())

            #put initial (dammy) values to arrays
            self.getInitialParameters()

            #share headers
            headerInitialized = False
            headerMatched = False
            while not headerMatched:
                self.socket.send_string("isHeaderInitialized()")
                headerInitialized = str2Bool(self.socket.recv_string())
                if headerInitialized:
                    #confirmHeaderMatching (present do nothing)
                    self.socket.send_string("sendParamHeaderAsStr()")
                    print(self.socket.recv_string())
                    self.socket.send_string("sendResultHeaderAsStr()")
                    print(self.socket.recv_string())
                    #if discrepancy of headers was found, headerInitialized flags should be turned off
                    #headerMatched = False
                    #else
                    headerMatched = True
                    print("Header matching completed.")
                else:
                    sendStr = "receiveParamHeaderAsStr(\"%s\")" % self.sendParamHeaderAsStr()
                    self.socket.send_string(sendStr)
                    print(self.socket.recv_string())
                    sendStr = "receiveResultHeaderAsStr(\"%s\")" % self.sendResultHeaderAsStr()
                    self.socket.send_string(sendStr)
                    print(self.socket.recv_string())
                    print("Headers are sent to learner.")
                time.sleep(WAITTIME)

            #confirm Learner is Ready
            learnerReady = False
            while not learnerReady:
                self.socket.send_string("isLearnerRunning()")
                learnerReady = str2Bool(self.socket.recv_string())
                print(learnerReady)
                time.sleep(WAITTIME)

            #Tell sequencer is running
            self.socket.send_string("sequencerRunning()")
            print(self.socket.recv_string())

            ##Main rootin
            while True:
                #send latest result and experimental parameters
                if self.lastExpReady:
                    self.socket.send_string("isLastExpUnread()")
                    lastExpUnread = str2Bool(self.socket.recv_string())
                    if not lastExpUnread:
                        sendStr = "receiveLastExpAsStr(\"%s\")" % self.sendLastExpAsStr()
                        self.socket.send_string(sendStr)
                        print("Last parameters to learner: " + self.socket.recv_string())

                #get new experimental parameters and do experiment
                if not self.nextExpReceived:
                    self.socket.send_string("isNextExpUnread()")
                    nextExpUnread = str2Bool(self.socket.recv_string())

                    if nextExpUnread:
                        self.socket.send_string("sendNextExpAsStr()")
                        recvStr = self.socket.recv_string()
                        print("Next paramters from leaner: " + recvStr)
                        self.receiveNextExpAsStr(recvStr)

                if self.isDammyExperimentReady():
                    self.doDammyExperiment(experiment)

                time.sleep(WAITTIME)

        finally:
            pass
