#!/bin/bash
gunicorn --bind=0.0.0.0 --workers=4 --timeout 600 app:app