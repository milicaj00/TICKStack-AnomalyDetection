from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
import logging
import json
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger()


class LOFHandler(Handler):
    
    def __init__(self, agent):
        logger.info('__init__ trigger')
        self._agent = agent
        self._field = ' '
        self._size = 10
        self._points = []
        self._state = {}

    def init(self, init_req):
        logger.info('INIT trigger')
        for opt in init_req.options:
            if opt.name == 'field':
                self._field = opt.values[0].stringValue
        success = True
        msg = ''
        if self._field == '':
            success = False
            msg = 'must provide field name'
        response = udf_pb2.Response()
        response.init.success = success
        response.init.error = msg.encode()
        return response
    
    def info(self):
        logger.info('info trigger')
        response = udf_pb2.Response()
        response.info.wants = udf_pb2.BATCH
        response.info.provides = udf_pb2.STREAM
        response.info.options['field'].valueTypes.append(udf_pb2.STRING)
        return response

    def begin_batch(self, begin_req):
        #self._points = []
        logger.info('begin_batch trigger')

    def snapshot(self):
        data = {}
        for group, state in self._state.items():
            data[group] = state.snapshot()
        response = udf_pb2.Response()
        response.snapshot.snapshot = json.dumps(data).encode()
        return response
        
    def restore(self, restore_req):
        logger.info('restore')
        response = udf_pb2.Response()
        response.init.success = False
        return response
        
    def point(self, point):
        logger.info('point trigger')
        self._points.append(point.fieldsDouble[self._field])
       

    def end_batch(self, batch_meta):
        logger.info(self._points)
        n_neighbors = 3
        X = np.array(self._points)

        reshaped_data =  np.array(self._points).reshape(-1,1)

        if len(self._points) > 13:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05)
            y_pred = lof.fit_predict(reshaped_data)

            response = udf_pb2.Response()
            response.point.time = batch_meta.tmax
            response.point.name = batch_meta.name
            response.point.group = batch_meta.group
            response.point.tags.update(batch_meta.tags)

            for i in range(len(y_pred)):
                if y_pred[i] == -1:
                    response.point.fieldsDouble['point'] = self._points[i]
                    self._agent.write_response(response)
            
            self._points.pop(0)

        logger.info('end_batch')


if __name__ == '__main__':

    agent = Agent()
    h = LOFHandler(agent)
    agent.handler = h

    logger.info("Starting agent for anomaly detection Handler")
    agent.start()
    agent.wait()
    logger.info("Agent finished")