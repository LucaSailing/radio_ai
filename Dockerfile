# Use the official slim image as the base
# the slim version is lighter and enough for this purposes
FROM python:3.10.6

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Set the working directory inside the container
WORKDIR /api

# Copy your requirements file
COPY requirements.txt requirements.txt

# Use the pre-installed pip to install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all of my package: by doing this stage after the pip installs
# they won't have to be reinstalled everytime I do a change to
# taxifare
COPY radio_ai_package radio_ai_package

# Set the default command
CMD uvicorn radio_ai_package.api.fast:app --host 0.0.0.0 --port $PORT


#### the last port parameter is to tell my container through which window
#### to connect with cloud
