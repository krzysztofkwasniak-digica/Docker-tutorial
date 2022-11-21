# Week 1
### Setup
```sh
cd week1/
```
### Training script

Let's look at the code in src/train.py

```python
from pathlib import Path
from typing import Final, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.logger import Logger

DATASET_PATH: Final = Path("data/healthcare-dataset-stroke-data.csv")


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(data_path)


def preprocess_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_no_idx: pd.DataFrame = raw_data.drop(columns=["id"])  # Drop Id column as it holds no useful information
    data_drop_unknown: pd.DataFrame = data_no_idx[data_no_idx["smoking_status"] != "Unknown"]
    data_no_na: pd.DataFrame = data_drop_unknown[data_drop_unknown["bmi"] != "N/A"].dropna()
    data_dummy: pd.DataFrame = pd.get_dummies(
        data_no_na,
        columns=["gender", "ever_married", "Residence_type", "smoking_status", "work_type"],
        drop_first=True,
    )
    data_dummy = data_dummy.rename(
        columns={
            "work_type_Self-employed": "work_type_Self_employed",
            "smoking_status_never smoked": "smoking_status_never_smoked",
        }
    )
    X: pd.DataFrame = data_dummy.drop(columns=["stroke"])
    y: pd.DataFrame = data_dummy["stroke"]
    return X, y


def main() -> None:
    logger = Logger(__name__)
    raw_data: pd.DataFrame = load_data(DATASET_PATH)
    X, y = preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    results: float = clf.score(X_test, y_test)
    logger.log.info(f"Accuracy: {results}")
    model_dir: Path = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_save_path: Path = model_dir / "model.joblib"
    joblib.dump(clf, model_save_path)
    logger.log.info(f"Model successfully saved to {model_save_path}")


if __name__ == "__main__":
    main()


```
---
## First task
The training script is ready. The question is how to dockerize it?
<details>
    <summary>What does "dockerize" mean?</summary>

> Dockerizing is the process of packing, deploying, and running applications  using Docker containers
> -- <cite> [https://developerexperience.io/practices/dockerizing](https://developerexperience.io/practices/dockerizing) </cite>
</details>

Lets view the contents of **`Dockerfile.train`**.

```dockerfile
FROM python:3.9-slim-buster

LABEL AUTHOR=krzysztof.kwasniak@digica.com

WORKDIR /app/

COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

COPY src/logger.py src/logger.py
COPY src/train.py src/train.py

ENTRYPOINT ["python", "src/train.py"]
```

There are a few things that require explanation. Let's quickly go through them.

#### FROM
```dockerfile
FROM python:3.9-slim-buster
```
Here, we specify the base image that we will use as a foundation. It is important to note, that the image does not have to exist locally. Docker will first search for it locally, but if the image does not exist, it will be pulled from the [Dockerhub repository](https://hub.docker.com/).
But why did I choose `python:3.9-slim-buster` as a base image? That is actually a valid question. The base image should satisfy all our needs, while consuming the least amount of memory.
[How should I pick the right Docker image?](https://stackoverflow.com/questions/57918880/how-should-i-pick-the-right-docker-image)
Any base image from the `ubuntu` repository would also be sufficient, however I would have to install python manually.
> <cite>[https://docs.docker.com/engine/reference/builder/#from](https://docs.docker.com/engine/reference/builder/#from)</cite>
<details>
    <summary>How would a ubuntu:20.04 base image look like?</summary>

```dockerfile
FROM ubuntu20:04
RUN apt-get update && apt-get install -y \
    python3.9-dev python3-pip
```

</details>

#### WORKDIR
```dockerfile
WORKDIR /app/
```
> The `WORKDIR` instruction sets the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY` and `ADD` instructions that follow it in the Dockerfile. If the `WORKDIR` doesn’t exist, it will be created even if it’s not used in any subsequent `Dockerfile` instruction.
> <cite>[https://docs.docker.com/engine/reference/builder/#workdir](https://docs.docker.com/engine/reference/builder/#workdir)</cite>

#### COPY
```dockerfile
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/logger.py src/logger.py
COPY src/train.py src/train.py
```
Here, we copy all the files that our container will need.
> The `COPY` instruction copies new files or directories from <src> and adds them to the filesystem of the container at the path <dest>.
<cite>[https://docs.docker.com/engine/reference/builder/#copy](https://docs.docker.com/engine/reference/builder/#copy)</cite>

#### RUN
```dockerfile
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .
```
We specify the commands that should be executed in the shell.
It is important to note, that these commands are written in a single `RUN`. The reason for that is because each `RUN` statement will create a new layer. This is crucial in building larger images.
[Multiple RUN vs. single chained RUN in Dockerfile, which is better?](https://stackoverflow.com/questions/39223249/multiple-run-vs-single-chained-run-in-dockerfile-which-is-better)
>The `RUN` instruction will execute any commands in a new layer on top of the current image and commit the results. The resulting committed image will be used for the next step in the Dockerfile.
<cite>[https://docs.docker.com/engine/reference/builder/#run](https://docs.docker.com/engine/reference/builder/#run)</cite>

#### ENTRYPOINT
Here we define what docker will execute when running the image.
```dockerfile
ENTRYPOINT ["python", "src/train.py"]
```
> An `ENTRYPOINT` allows you to configure a container that will run as an executable.
<cite>[https://docs.docker.com/engine/reference/builder/#entrypoint](https://docs.docker.com/engine/reference/builder/#entrypoint)</cite>


### Building the image
Execute the command in your terminal to [build](https://docs.docker.com/engine/reference/commandline/build/) the base image.
```bash
docker build .
```
> ~/Docker-tutorial/week1$ docker build .
unable to prepare context: unable to evaluate symlinks in Dockerfile path: lstat /home/kkwasniak/Docker-tutorial/week1/Dockerfile: no such file or directory

Oops, looks like it didn't work. Let's examine why.
By default, Docker looks for a `Dockerfile` in the specified context (the `"."` in `docker build .` refers to the current directory `~/Docker-tutorial/week1`). We have two `Dockerfiles`.
- **`Dockerfile.train`**
- **`Dockerfile.app`**

So, to build the image, we must specify the path to the correct `Dockerfile`!
We can do that by passing the `-f` option to the `build` command.
> -f, --file string             Name of the Dockerfile (Default is 'PATH/Dockerfile')

Let's try.
```sh
docker build -f Dockerfile.train .
```
If everything went well you should see a similar output:
> Successfully built 4f7f85ad2bc5

Congratulations! You have built the training image! Let's see it on the docker image list.
```sh
docker images
```
> ~/Docker-tutorial/week1$ docker images
REPOSITORY   TAG                  IMAGE ID       CREATED         SIZE
<none>       <none>               4f7f85ad2bc5   4 minutes ago   658MB

The image is built. But how do I remember the `IMAGE ID`? There must be a better way to version the images, right? Correct! Usually, when building an image you pass a tag to distinguish it from the rest of the images you have built. How do we do that? By passing another option `-t` to the `build` command.
> -t, --tag list                Name and optionally a tag in the 'name:tag' format

See for yourself.
```sh
docker build -t train:latest -f Dockerfile.train .
docker images
```
> ~/Docker-tutorial/week1$ docker images
REPOSITORY   TAG                  IMAGE ID       CREATED         SIZE
train        latest               4f7f85ad2bc5   9 minutes ago   658MB

Docker recognizes that the image we are trying to build is the same as the image we have built before, just with a tag so it just tags the existing image. It is equal to running
```sh
docker tag 4f7f85ad2bc5 train:latest
```

Everything is now built!

### Running the image

Thanks to the tagging we can now refer to our image by the repository:tag. Let's run it.

```sh
docker run train:latest
```

But once again, we run into an error.
> FileNotFoundError: [Errno 2] No such file or directory:
'data/healthcare-dataset-stroke-data.csv'

Looks like we forgot to add `COPY data data` to the `Dockerfile.train`, right?
![](https://i.kym-cdn.com/entries/icons/original/000/028/596/dsmGaKWMeHXe9QuJtq_ys30PNfTGnMsRuHuo_MUzGCg.jpg)

Copying the data into the image is a bad practice. A container can be stopped, destroyed or replaced. This should be done without any impact or loss of data. But our training script requires the data to be inside the container. How to inject it? By using [volumes](https://docs.docker.com/storage/volumes/).
Lets mount the **`data`** directory from the host to the container.
```sh
docker run -v "$(pwd)"/data:/app/data train:latest
```
>~Docker-tutorial/week1 docker run -v "$(pwd)"/data:/app/data train:latest
[01/30/22 14:29:28] INFO     train:__main__: 46 - Accuracy: 0.9406614785992218         train.py:46
                    INFO     train:__main__: 51 - Model successfully saved to models/model.joblib train.py:51

But if we look inside the **`models`** directory, we will only find **`.gitkeep`**.
```sh
ls models/ -a
```
That is because we didn't mount the model directory to the container. The model was saved inside the container. Fix it by mounting the model directory.
```sh
docker run -v "$(pwd)"/data:/app/data -v "$(pwd)"/models:/app/models train:latest
```
You might run into a permission issue.
>PermissionError: [Errno 13] Permission denied: 'models/model.joblib'

Fix it by adding all permission to the folder and running the command again.
```sh
chmod 777 models/
docker run -v "$(pwd)"/data:/app/data -v "$(pwd)"/models:/app/models train:latest
```

If you don't want to see the output add a `-d` option to the `run` command. It will not occupy your terminal. This is useful for scripts with very long output.
```sh
docker run -d -v "$(pwd)"/data:/app/data -v "$(pwd)"/models:/app/models train:latest
```
> ~/Docker-tutorial/week1 docker run -d -v "\$(pwd)"/data:/app/data -v "\$(pwd)"/models:/app/models train:latest
7bf21868cfea2cd49c306b144e1197ed52b479e3759676a04aebc23f943e2a8f

To see if everything went ok check the logs.
```sh
docker logs 7bf21868cfea2cd49c306b144e1197ed52b479e3759676a04aebc23f943e2a8f
```

If you lose the ID, you can check it with
```sh
docker ps -a
```
Which will show you all containers. You can then copy the `CONTAINER ID`.

---

## <center>HOMEWORK</center>
The file **`src/app.py`** contains the code needed to run a simple FastAPI application to serve the previously trained model.
Your homework is to fill in the missing code in the **`Dockerfile.app`** and build the image. Think of the following things:
- Which base image should be chosen for this app?
- What files do you need to copy?
- What dependencies does the **`src/app.py`** have?
- What are the shell commands that you need to run?
- How to specify the entrypoint? Refer to the `start_api.sh`.

Once you do it:
- Run the container in an interactive mode (`-it`). Play with the application in your browser (Try it out in the top left corner and execute with the example params).
    ```sh
    docker run -it -v "$(pwd)"/models:/app/models -p 8888:8888 app:latest
    ```
    [https://www.whitesourcesoftware.com/free-developer-tools/blog/docker-expose-port/](https://www.whitesourcesoftware.com/free-developer-tools/blog/docker-expose-port/)


- Run the container in a detached mode. Check if the container started. Attach into the running container. Play with the application in your browser.
    ```sh
    docker run -d -v "$(pwd)"/models:/app/models -p 8888:8888 app:latest
    docker ps
    docker attach "YOUR CONTAINER ID"
    ```
- Run the container in a detached mode. Enter into it. Look at the folder structure.
    ```sh
    docker run -d -v "$(pwd)"/models:/app/models -p 8888:8888 app:latest
    docker exec -it "YOUR CONTAINER ID" bash
    ls
    ```
- Run the container in a detached mode and then stop it.
    ```sh
    docker run -d -v "$(pwd)"/models:/app/models -p 8888:8888 app:latest
    docker stop "YOUR CONTAINER ID"
    ```
- Override the containers entrypoint. Start the application manually.
    ```sh
    docker run --entrypoint bash -it -v "$(pwd)"/models:/app/models -p 8888:8888 app:latest
    uvicorn app:app --reload --app-dir=src --host "0.0.0.0" --port 8888
    ```

## <center>That's it for week 1!</center>

------
### Additional read

https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

https://docs.docker.com/develop/dev-best-practices/

https://sysdig.com/blog/dockerfile-best-practices/

https://testdriven.io/blog/docker-best-practices/

https://docs.docker.com/engine/reference/run/

https://docs.docker.com/engine/reference/commandline/build/

https://phoenixnap.com/kb/docker-cmd-vs-entrypoint

https://blog.logrocket.com/docker-volumes-vs-bind-mounts/

https://github.com/wsargent/docker-cheat-sheet/blob/master/README.md#best-practices





