# Full tag list: https://hub.docker.com/r/library/kapacitor/tags/
FROM kapacitor:latest

 RUN apt-get update && apt-get install -y python3.9 python3-pip 
 
 #RUN python get-pip.py
 
#RUN apt-get -qq update && apt-get -qq install -y python3.9  python3-pip 


 
# Install python dependencies
RUN pip install \
	numpy \
	pandas \
	scipy


#RUN pip install google-api-python-client
#RUN pip install google
#RUN pip install protobuf==3.20.0 

# Install git
#RUN apt-get -qq install -y git

# Install kapacitor agent
#RUN git clone https://github.com/influxdata/kapacitor.git /tmp/kapacitor


# Start kapacitor service (log-level: debug | info | error)
#CMD ["kapacitord","-log-level","debug"]