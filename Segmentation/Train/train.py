import wandb
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import time
from wandb.keras import WandbCallback


def train_cycle(
        project_name,
        config,
        model,
        train_dataset,
        valid_dataset,
        train_size,
        logger,
        wandb_key=None,
        output_path=None
):
    checkpoint_name = str(time.time()) + f"_{config['model']}_" + config['dataset_name']
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(project=project_name, entity="ivanvykopal")

        config['model_name'] = checkpoint_name
        wandb.config.update(config)

    print(model.summary())

    if output_path:
        checkpoint_path = f"{output_path}/{checkpoint_name}"
    else:
        checkpoint_path = f"./{checkpoint_name}"

    mcp_save = ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.001
    )

    if wandb_key:
        callbacks = [mcp_save, WandbCallback(), reduce_lr, logger]
    else:
        callbacks = [mcp_save]

    model.fit(
        train_dataset,
        epochs=config['epochs'],
        validation_data=valid_dataset,
        steps_per_epoch=int(train_size // config['batch_size']),
        callbacks=callbacks
    )

    return checkpoint_name
