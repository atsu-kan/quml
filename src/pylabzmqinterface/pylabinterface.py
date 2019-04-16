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


import json
from typing import Any, Callable, Optional

import numpy as np
import zmq

from .adapter import Adapter


class NumpyEncoder(json.JSONEncoder):
    """ Extends JSONEncoder to serialize numpy arrays.
    To use this encoder: json.dumps(<numpy_array>, cls=NumpyEncoder)

    http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    """

    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict
        holding dtype, shape and the data.

        :param obj: object to be encoded
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_json = obj_data.tolist()
            # data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_json, dtype=str(obj.dtype), shape=obj.shape)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(self, obj)


class PyLabInterface(object):
    def __init__(self, binder) -> None:
        #flags for sequencer state control
        self.sequencerReady = False
        #flags for learner state control
        self.learnerReady = False
        #headers (arrays storing name of parameters/results)
        self.paramHeader = np.array([""])
        self.resultHeader = np.array([""])
        self.paramHeaderReceived = False
        self.resultHeaderReceived = False
        #Last experiment (parameters and results) from experimenter
        self.lastExpParam = np.array([0], dtype = float)
        self.lastExpResult = np.array([0], dtype = float)
        self.lastExpUnread = False
        #Next parameters to experimenter
        self.nextExpParam = np.array([0], dtype = float)
        self.nextExpUnread = False
        #zeroMQ socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(binder)

        #initialize state flags manually
    def reInitialize(self):
        self.sequencerReady = False
        self.learnerReady = False
        self.paramHeaderReceived = False
        self.resultHeaderReceived = False
        self.lastExpUnread = False
        self.nextExpUnread = False
        return "Re-initialized."

        #sequencer state control
    def sequencerRunning(self):
        self.sequencerReady = True
        return "Sequencer get ready."
    def sequencerStopped(self):
        self.sequencerReady = False
        return "Sequencer stopped."
    def isSequencerRunning(self):
        return self.sequencerReady
        #learner state control
    def learnerRunning(self):
        self.learnerReady = True
        return "Learner get ready."
    def learnerStopped(self):
        self.learnerReady = False
        return "Learner stopped."
    def isLearnerRunning(self):
        return self.learnerReady

        #methods for ZMQ socket
    def solveRequest(self):
        #  Wait for request from client
        message = self.socket.recv_string()
        print("Received request: %s" % message)
        offsetStr = "self."
        message = offsetStr + message
        try:
            r = eval(message)
            print(("Return value: ", r))
            print(("Type: ", type(r)))
            self.socket.send_string(json.dumps(r, cls=NumpyEncoder))
        except NameError:
            print("except NameError")
            self.socket.send_string("Unknown command")
        except SyntaxError:
            print("except SyntaxError")
            self.socket.send_string("Invalid syntax")
        except:
            print("except")
            self.socket.send_string("Unknown error")

        #methods for headers (name of parameters/results)
    def isHeaderInitialized(self):
        return self.paramHeaderReceived and self.resultHeaderReceived

    def getParamHeader(self):
        return self.paramHeader

    def getResultHeader(self):
        return self.resultHeader

    def receiveParamHeaderAsStr(self, paramHeaderStr):
        self.paramHeader = np.array(paramHeaderStr.split("\t"))
        self.paramHeaderReceived = True
        return self.sendParamHeaderAsStr()

    def sendParamHeaderAsStr(self):
        headerStr = ""
        for hstr in self.paramHeader:
            headerStr += hstr
            headerStr += "\t"
        return headerStr.rstrip("\t")

    def receiveResultHeaderAsStr(self, resultHeaderStr):
        self.resultHeader = np.array(resultHeaderStr.split("\t"))
        self.resultHeaderReceived = True
        return self.sendResultHeaderAsStr()

    def sendResultHeaderAsStr(self):
        headerStr = ""
        for hstr in self.resultHeader:
            headerStr += hstr
            headerStr += "\t"
        return headerStr.rstrip("\t")

        #methods for last Experiment IO
    def isLastExpUnread(self):
        return self.lastExpUnread

    def getLastParam(self):
        return self.lastExpParam

    def getLastResult(self):
        self.lastExpUnread = False
        return self.lastExpResult

    def receiveLastExpAsStr(self, paramStr, resultStr):
        self.lastExpParam = self.str2NParray(paramStr)
        self.lastExpResult = self.str2NParray(resultStr)
        self.lastExpUnread = True
        return self.npArray2Str(self.lastExpParam)

        #methods for next Experiment IO
    def isNextExpUnread(self):
        return self.nextExpUnread

    def receiveNextExp(self, expArray):
        self.nextExpParam = expArray
        self.nextExpUnread = True

    def sendNextExpAsStr(self):
        self.nextExpUnread = False
        return self.npArray2Str(self.nextExpParam)

    @staticmethod
    def npArray2Str(npArray):
        npAsStr = ""
        for num in npArray:
            npAsStr += f'{num:.12f}'
            npAsStr += "\t"
        return npAsStr.rstrip("\t")

    #slice the camma separated array str into nparray
    @staticmethod
    def str2NParray(arrayStr):
        #print arrayStr.split(",")
        list = []
        for valStr in arrayStr.split("\t"):
            try:
                valNum = float(valStr)
            except:
                valNum = float("NaN")
            list.append(valNum)
        npArray = np.array(list)
        return npArray

    @classmethod
    def run(cls, binder: Any, create_adapter: Callable[[np.ndarray, np.ndarray, np.ndarray], Adapter]) -> None:

        adapter: Optional[Adapter] = None

        #make PyLabInterface instance
        self = cls(binder)
        try:
            self.learnerStopped()

            #initialize learner's parameter
            isNextExperimentReady = False
            lastExperiment = self.getLastParam()

            #wait initialize of headers
            while not self.isHeaderInitialized():
                self.solveRequest()

            #Store headers and Learner get ready
            paramHeader = self.getParamHeader()
            resultHeader = self.getResultHeader()
            self.learnerRunning()

            #wait sequencer get ready
            while not self.isSequencerRunning():
                self.solveRequest()

            #main rootin
            while True:
                #Catch the latest experiment
                if (not isNextExperimentReady) and self.isLastExpUnread():
                    lastExperiment = self.getLastParam()
                    lastResult = self.getLastResult()
                    if adapter is None:
                        adapter = create_adapter(paramHeader, resultHeader, lastExperiment)
                    else:
                        adapter.write(lastExperiment, lastResult)
                    isNextExperimentReady = True

                #echo back
            #    if isNextExperimentReady and (not interface.isNextExpUnread()):
            #        interface.receiveNextExp(lastExperiment)
            #        isNextExperimentReady = False
                #increment dammy index and put random parameters
                if isNextExperimentReady and (not self.isNextExpUnread()) and adapter is not None:
                    isLearnerRunning, nextExperiment = adapter.read()
                    if not isLearnerRunning:
                        break
                    if nextExperiment is not None:
                        self.receiveNextExp(nextExperiment)
                        isNextExperimentReady = False

                #handle the request from experimenter
                self.solveRequest()

            #interface.learnerStopped()
            #print "done"
        finally:
            if adapter is not None:
                adapter.shutdown()
