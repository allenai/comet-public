To run a generation experiment (either conceptnet or atomic), follow these instructions:


<h1>Installing Dependencies</h1>

First clone, the repo:

```
git clone https://github.com/allenai/comet-public.git
cd comet
```

Then run the setup scripts to acquire the pretrained model files from OpenAI, as well as the ATOMIC and ConceptNet datasets

```
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
```

Then install dependencies (assuming you already have Python 3.6 and Pytorch >= 1.0:

```
pip install -r requirements.txt
python -m spacy download en
```

<h1> Installing the Package </h1>

You should now be able to use most COMeT functionality!

<h1> Launching a demo </h1>

First, download the pretrained models from the following link:

```
wget https://storage.googleapis.com/ai2-mosaic/public/comet/models.zip
unzip models.zip
```


Then to launch the demo, do the following:

```
from comet.interactive.atomic_demo import DemoModel

demo_model = DemoModel("/path/to/pretrained_model")

demo_model.predict("PersonX goes to the mall", "xEffect", "beam-10")

```

Or for ConceptNet

```
from comet.interactive.conceptnet_demo import DemoModel

demo_model = DemoModel("/path/to/pretrained_model")

demo_model.predict("man with axe", "CapableOf", "beam-10")

```
