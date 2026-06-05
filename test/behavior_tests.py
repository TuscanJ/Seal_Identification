from behave import given, when, then


@given("a user submits a photo of a seal")
def step_given_user_submits_photo(context):
    pass


@when("the total confidence for the top three seals is 0.45")
def step_when_confidence_045(context):
    pass


@then("a text popup will appear on the screen alerting that the seal may be new")
def step_then_popup_appears(context):
    pass


@then("the dropdown menu to select a seal will include a new seal option")
def step_then_dropdown_has_new_option(context):
    pass


@when("the total confidence for the top three seals is 0.76")
def step_when_confidence_076(context):
    pass


@then("a text popup will not appear on the screen alerting that the seal may be new")
def step_then_popup_does_not_appear(context):
    pass


@when("the total confidence for the top three seals is 0.02")
def step_when_confidence_002(context):
    pass


@when("the total confidence for the top three seals is 3.00")
def step_when_confidence_300(context):
    pass


@given("a photo is of a seal known to not be in the database")
def step_given_unknown_seal_photo(context):
    pass


@when("that photo is submitted to the seal classifier")
def step_when_photo_submitted_to_classifier(context):
    pass


@given("a photo is entirely black pixels")
def step_given_black_image(context):
    pass
