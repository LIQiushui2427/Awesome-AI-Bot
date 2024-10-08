# Use an unofficial Python runtime as a parent image
FROM cnstark/pytorch:1.12.0-py3.9.12-cuda11.6.2-ubuntu20.04

RUN apt-get update && apt-get install

RUN apt-get install -y locales && locale-gen en_US.UTF-8 
#3&& update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 LANGUAGE=zh_CN.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8

# Set the working directory to /app

# Copy the current directory contents into the container at /app


WORKDIR /app

COPY . /app

# uncommend this line if you want to use fix volume
# VOLUME ["/app/data", "/app/outputsByAI", "/app/outputsByBt"]

# Install any needed packages specified in requirements.txt

RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container


EXPOSE 80


# CMD ["python", "app.py"]

CMD ["python", "app.py"]



# Define environment variable
# ENV NAME World
