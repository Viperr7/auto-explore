import io

from PIL import Image

from harness.repair import Budget, Frame, action_point, same_state


def frame(label="Details", *, duplicate=False, selected="false", package="test.app"):
    png = io.BytesIO()
    Image.new("RGB", (100, 180)).save(png, format="PNG")
    node = (f'<node package="{package}" resource-id="{package}:id/detail" text="{label}"'
            f' class="android.widget.Button" clickable="true" enabled="true" selected="{selected}" bounds="[4,12][30,36]"/>')
    return Frame(f"{package}/.Main", png.getvalue(), f'<hierarchy>{node}{node if duplicate else ""}</hierarchy>'.encode())


def test_state_and_live_selector():
    source = frame()
    assert same_state(source, frame(), "test.app")
    assert not same_state(source, frame(selected="true"), "test.app")
    selector = {"resource_id": "test.app:id/detail", "label": "Details"}
    assert action_point(source, selector, "test.app") == (17, 24)
    assert action_point(frame(duplicate=True), selector, "test.app") is None
    assert action_point(frame(package="other.app"), selector, "test.app") is None
    assert action_point(frame("Pay"), {"resource_id": "test.app:id/detail", "label": "Pay"}, "test.app") is None
    obstruction = b'<node package="test.app" resource-id="test.app:id/pay" text="Pay" clickable="true" bounds="[10,15][40,40]"/>'
    crowded = Frame(source.activity, source.png, source.xml.replace(b"</hierarchy>", obstruction + b"</hierarchy>"))
    assert action_point(crowded, selector, "test.app") is None


def test_budget_counts_failed_attempts():
    budget = Budget(actions=1, seconds=60)
    assert budget.invoke(lambda: False) is False
    assert budget.used == 1
    try:
        budget.invoke(lambda: (_ for _ in ()).throw(AssertionError("executed after budget")))
    except TimeoutError:
        pass
    else:
        raise AssertionError("expected budget exhaustion")
