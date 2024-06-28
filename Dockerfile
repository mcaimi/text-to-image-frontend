FROM registry.access.redhat.com/ubi9/python-311:1-62

# By default, listen on port 8080
EXPOSE 8000/tcp
ENV PORT=8000
ENV INFER_URL="localhost/v1/models/model:predict"
ENV MODEL_NAME="DreamShaper_8_pruned.safetensors"
ENV RUN_LOCALLY="False"

# Set the working directory in the container
WORKDIR /projects

# Copy the content of the local src directory to the working directory
COPY . .

# Install any dependencies
RUN \
  if [ -f requirements.txt ]; \
    then pip install -r requirements.txt; \
  elif [ `ls -1q *.txt | wc -l` == 1 ]; \
    then pip install -r *.txt; \
  fi

# Specify the command to run on container start
ENTRYPOINT ["fastapi", "run", "./frontend.py"]
