# Implementation: Oleh Bakumenko, Univerity of Duisburg-Essen

import sys
sys.path.append("../")
import time
import torch, torch.nn as nn
from utility import utils as uu
from utility.eval import evaluate_classifier_model
from utility.confusion_matrix import calculate_confusion_matrix


def training_loop(epochs, optimizer, model, criterion, ds, dl, batch_size, run_name, device, cache_me=False, mod_step=500, time_me=True, time=time):
    num_steps = len(ds.file_names['train']) // batch_size

    for epoch in range(epochs):

        # If we are caching, we now have all data and let the (potentially non-persistent) workers know
        if cache_me is True and epoch > 0:
            dl.dataset.set_cached("train")
            dl.dataset.set_cached("val")

        # Time me
        if time_me is True:
            e_start = time.time()

        # Go to train mode
        ds.set_mode("train")
        model.train()

        # Train loop
        for step, (data, targets) in enumerate(dl):

            # Manually drop last batch (this is for example relevant with BatchNorm)
            if step == num_steps - 1 and (epoch > 0 or ds.cache_data is False):
                continue

            # Train loop: Zero gradients, forward step, evaluate, log, backward step
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            loss = criterion(predictions, targets)
            if step % mod_step == 0:
                print(f"Epoch [{epoch + 1}/{epochs}]\t Step [{step + 1}/{num_steps}]\t Train Loss: {loss.item():.4f}")
            uu.csv_logger(
                logfile=f"../logs/{run_name}_train.csv",
                content={"epoch": epoch, "step": step, "loss": loss.item()},
                first=(epoch == 0 and step == 0),
                overwrite=(epoch == 0 and step == 0)
            )
            loss.backward()
            optimizer.step()

        # Go to eval mode
        ds.set_mode("val")
        model.eval()

        # Validation loop
        val_accuracy, avg_val_loss = evaluate_classifier_model(model=model, dataloader=dl, device=device)
        print(f"Epoch [{epoch + 1}/{epochs}]\t Val Loss: {avg_val_loss:.4f}\t Val Accuracy: {val_accuracy:.4f}")
        uu.csv_logger(
            logfile=f"../logs/{run_name}_val.csv",
            content={"epoch": epoch, "val_loss": avg_val_loss, "val_accuracy": val_accuracy},
            first=(epoch == 0),
            overwrite=(epoch == 0)
        )

        if time_me is True:
            epoch_time = (time.time() - e_start)
            uu.csv_logger(
                logfile=f"../logs/{run_name}_runtime.csv",
                content={"epoch": epoch, "runtime": epoch_time},
                first=(epoch == 0),
                overwrite=(epoch == 0)
            )
            print(f"Epoch nr {epoch + 1} runtime: {epoch_time:.4f}s")

    # Finally, test time
    ds.set_mode("test")
    model.eval()

    test_accuracy, avg_test_loss = evaluate_classifier_model(model=model, dataloader=dl, device=device)
    print(f"Epoch [{epoch + 1}/{epochs}]\t Test Loss: {avg_test_loss:.4f}\t Test Accuracy: {test_accuracy:.4f}")
    uu.csv_logger(
        logfile=f"../logs/{run_name}_test.csv",
        content={"epoch": epoch, "test_loss": avg_test_loss, "test_accuracy": test_accuracy},
        first=True,
        overwrite=True
    )


def training_loop_conf_matr(epochs, optimizer, model, criterion, ds, dl, batch_size, run_name, device, time_me=True, time=time):
    conf_matr_save = torch.zeros(epochs + 1, 3, 3)
    per_class_accuracy_save = torch.zeros(epochs + 1, 3)
    model = model.to(device)
    num_steps = len(ds.file_names['train']) // batch_size
    train_start = time.time()

    for epoch in range(epochs):
        print(f"Time_elapsed: {(time.time() - train_start) / 60 :.2f} min")

        # Time me
        if time_me is True:
            e_start = time.time()

        # Go to train mode
        ds.set_mode("train")
        model.train()

        # Train loop
        for step, (data, targets) in enumerate(dl):

            # Manually drop last batch (this is for example relevant with BatchNorm)
            if step == num_steps - 1 and (epoch > 0 or ds.cache_data is False):
                continue

            # Train loop: Zero gradients, forward step, evaluate, log, backward step
            optimizer.zero_grad()
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            loss = criterion(predictions, targets)
            uu.csv_logger(
                logfile=f"../logs/{run_name}_train.csv",
                content={"epoch": epoch, "step": step, "loss": loss.item()},
                first=(epoch == 0 and step == 0),
                overwrite=(epoch == 0 and step == 0)
            )
            loss.backward()
            optimizer.step()

        # Go to eval mode
        ds.set_mode("val")
        model.eval()

        # Validation loop
        val_accuracy, avg_val_loss = evaluate_classifier_model(model=model, dataloader=dl, device=device)
        confusion_matrix, acc = calculate_confusion_matrix(model=model, dataloader=dl, device=device)
        conf_matr_save[epoch] = confusion_matrix
        per_class_accuracy_save[epoch] = acc
        print(
            f"Epoch [{epoch + 1}/{epochs}]\t Val Loss: {avg_val_loss:.4f}\t Val Accuracy: {val_accuracy:.4f}, \nConfusion Matrix: \n{confusion_matrix}, \nPer-class Accuracy: {acc}, Mean : {acc.mean()} ")
        uu.csv_logger(
            logfile=f"../logs/{run_name}_val.csv",
            content={"epoch": epoch, "val_loss": avg_val_loss, "val_accuracy": val_accuracy},
            first=(epoch == 0),
            overwrite=(epoch == 0)
        )

    # Finally, test time
    ds.set_mode("test")
    model.eval()
    confusion_matrix_test, acc_test = calculate_confusion_matrix(model=model, dataloader=dl, device=device)
    conf_matr_save[epochs] = confusion_matrix_test
    per_class_accuracy_save[epochs] = acc_test
    print(f"Test Confusion Matrix: \n{confusion_matrix_test}, Per-class Accuracy: {acc_test}, Mean : {acc_test.mean()}")
    test_accuracy, avg_test_loss = evaluate_classifier_model(model=model, dataloader=dl, device=device)
    print(f"Test Loss: {avg_test_loss:.4f}\t Test Accuracy: {test_accuracy:.4f}")
    uu.csv_logger(
        logfile=f"../logs/{run_name}_test.csv",
        content={"epoch": epoch, "test_loss": avg_test_loss, "test_accuracy": test_accuracy},
        first=True,
        overwrite=True
    )
    torch.save(conf_matr_save, f='confusion_matr_save_' + run_name + '.pt')
    torch.save(per_class_accuracy_save, f='per_class_accuracy_save_' + run_name + '.pt')