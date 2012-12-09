from __future__ import division
import scaffold

class SprinklerState(scaffold.State):
    __slots__ = ['cloudy', 'sprinkler', 'rain']


class SprinklerChain(scaffold.Chain):
    def transition(self, state):
        pass

class SprinklerData(scaffold.DataSource):
    pass

expt = scaffold.Experiment()

