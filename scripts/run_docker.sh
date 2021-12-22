docker container run -itd \
    -p 7860:7860 \
    -v $PWD/src:/project/src \
    -v $PWD/weights:/project/weights \
    --name gradio_dev gradio_sample_dev bash

