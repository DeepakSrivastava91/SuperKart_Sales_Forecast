# Base image with Python 3.9
FROM python:3.9

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip3 install -r requirements.txt

# Run as a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Launch the Streamlit app on the port declared in README (app_port: 8501)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false"]
