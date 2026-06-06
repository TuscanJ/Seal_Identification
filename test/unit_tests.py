import pytest
from pytest_bdd import scenarios, given, when, then, parsers

# these are commented out for this check in to spare you the trouble of setting up a conda environment and reconciling outdated imports in old files
# import os
# import sys

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.append(parent_dir)

# from src import infer
# from src import hf_model
# from src import hf_dataset
# from app import app


def test_new_seal_popup_logic(confidence_score):
    #result = new_seal_popup(confidence_score). <- Not implemented yet (will test values under .75)
    result = .70
    popup_status = True if result < 0.75 else False
    assert popup_status is True

def new_seal_added(training_queue):
    #assert available_id is training_queue.id + 1 <- Not implemented yet but training_queue should unpack into id and class, where id = 512, and the next available id for new seals should be that + 1
    #assert class_new is training_queue.class <- and class should be new or old. old can be directly added to database but still flagged for training queue with a certain id.
    assert 0 == 0

def old_seal_added(training_queue):
    #assert class_old is training_queue.class <- same logic as above
    #assert training_queue.id in used_ids <- check id is in used list
    assert 0 == 0
