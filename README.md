# Style Transfer Website

A website that applies artistic styles to your photo. Based on the Artistic Style Transfer using Neural Networks.

## Quick Start

### Running locally

1. Clone the repository

    ```bash
    git clone https://github.com/keertirajmalik/neural.git
    cd neural
    ```

2. Create new virtual environment
    - On windows:

        ```bash
        py -3 -m venv venv
        ```

    - On Linux:

        ```bash
        python3 -m venv venv
        ```

3. Activate virtual environment
    - On windows:

        ```bash
        venv\Scripts\activate
        ```

    - On Linux:

        ```bash
        . venv/bin/activate
        ```

4. Install all the dependencies

    ```bash
    pip install -r requirements.txt
    ```

5. Run Web App

    ```bash
    gunicorn --bind 0.0.0.0:5000 wsgi --timeout 240
    ```


## Libraries used

- matplotlib
- scipy
- numpy
- Flask
- torch
- Pillow
- torchvision

## References

- [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)
- [Instance Normalization](https://arxiv.org/pdf/1607.08022.pdf)
