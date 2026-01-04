# Docker Deployment Guide

## Overview
This guide explains how to build and run the Obesity ML Application using Docker.

## Prerequisites
- Docker installed on your system
- Docker Compose (optional, but recommended)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Build and start the container:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Web UI: http://localhost:5000
   - Health check: http://localhost:5000/health
   - API prediction: http://localhost:5000/api/predict
   - API training: http://localhost:5000/api/train

3. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Commands

1. **Build the Docker image:**
   ```bash
   docker build -t obesity-ml-app .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     --name obesity-ml-app \
     -p 5000:5000 \
     -v $(pwd)/artifacts:/app/artifacts \
     -v $(pwd)/logs:/app/logs \
     obesity-ml-app
   ```

3. **View logs:**
   ```bash
   docker logs -f obesity-ml-app
   ```

4. **Stop the container:**
   ```bash
   docker stop obesity-ml-app
   docker rm obesity-ml-app
   ```

## Volume Mounts

The application uses two persistent volumes:

- **`./artifacts`**: Stores trained models and preprocessors
- **`./logs`**: Stores application logs

These volumes ensure that your models and logs persist even when the container is stopped or removed.

## Environment Variables

You can customize the application by setting environment variables:

```bash
docker run -d \
  -p 5000:5000 \
  -e FLASK_APP=app.py \
  -e PYTHONUNBUFFERED=1 \
  obesity-ml-app
```

## Health Check

The application includes a health check endpoint at `/health` that monitors:
- Application status
- Model availability
- Preprocessor availability

## Testing the API

### Prediction API
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 25,
    "Height": 1.75,
    "Weight": 80,
    "FCVC": 2,
    "NCP": 3,
    "CH2O": 2,
    "FAF": 1,
    "TUE": 0,
    "Gender": "Male",
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "SCC": "no",
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation"
  }'
```

### Training API
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{"use_hyperparameter_tuning": false}'
```

## Troubleshooting

### Container won't start
- Check logs: `docker logs obesity-ml-app`
- Ensure port 5000 is not already in use
- Verify all required files are present

### Model not found errors
- Ensure the artifacts directory exists
- Train the model first using the `/train` endpoint
- Check volume mounts are correctly configured

### Permission issues
- Ensure the artifacts and logs directories have correct permissions:
  ```bash
  chmod -R 755 artifacts logs
  ```

## Development Mode

For development with hot-reload:

```bash
docker run -d \
  -p 5000:5000 \
  -v $(pwd):/app \
  -e FLASK_ENV=development \
  obesity-ml-app
```

## Production Considerations

For production deployment:

1. **Use a production WSGI server** (modify Dockerfile CMD):
   ```dockerfile
   CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
   ```
   Add `gunicorn` to requirements.txt

2. **Remove debug mode**: Set `debug=False` in app.py

3. **Use environment variables** for sensitive configuration

4. **Set up proper logging** and monitoring

5. **Use Docker secrets** for sensitive data

## Cleaning Up

Remove all containers, images, and volumes:
```bash
docker-compose down -v
docker rmi obesity-ml-app
```
