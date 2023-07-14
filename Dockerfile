# Use an official Python runtime as a parent image
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

RUN apt-get update && apt-get install

RUN useradd -m lqs

RUN chown -R lqs:lqs /home/lqs

COPY --chown=lqs . /home/lqs

USER lqs

# Install any needed packages specified in requirements.txt
RUN cd /home/lqs/ && pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# WORKDIR /home/lqs

# Make po# EXPOSE 80

# Define environment variable
# ENV NAME World

# Run app.py when the container launches
# CMD ["python", "app.py"]
