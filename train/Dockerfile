FROM python:3.10-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set a custom token for Jupyter Notebook
ENV JUPYTER_TOKEN="default-jupyter-token"

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY . .

EXPOSE 8888

# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", \
#     "--allow-root", "--IdentityProvider.token=$JUPYTER_TOKEN"]
CMD ["/bin/bash", "run.sh"]
