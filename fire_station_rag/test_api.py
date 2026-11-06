import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from api import create_app


class ApiChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.calls = []

        def respond(question, history):
            self.calls.append((question, [m.content for m in history]))
            if question == "fail":
                raise RuntimeError("private provider detail")
            return "reply: " + question

        self.app = create_app(lambda: respond, Path(self.tmp.name) / "sessions.db", "test-secret")
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.headers = {"X-API-Key": "test-secret"}

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.tmp.cleanup()

    def session(self):
        response = self.client.post("/sessions", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        return response.json()["session_id"]

    def send(self, sid, message):
        return self.client.post(f"/sessions/{sid}/chat", headers=self.headers, json={"message": message})

    def test_auth_and_validation(self):
        self.assertEqual(self.client.post("/sessions").status_code, 401)
        self.assertEqual(self.send(str(uuid4()), "hello").status_code, 404)
        self.assertEqual(self.send(self.session(), " ").status_code, 422)
        self.assertEqual(self.send(self.session(), "x" * 4001).status_code, 422)

    def test_session_isolation_and_delete(self):
        first, second = self.session(), self.session()
        self.assertEqual(self.send(first, "hello").json()["answer"], "reply: hello")
        self.send(first, "follow up")
        self.assertEqual(self.calls[-1][1], ["hello", "reply: hello"])
        self.send(second, "new conversation")
        self.assertEqual(self.calls[-1][1], [])
        self.assertEqual(self.client.delete(f"/sessions/{first}", headers=self.headers).status_code, 204)
        self.assertEqual(self.send(first, "hello").status_code, 404)

    def test_failure_does_not_save_history_or_leak_error(self):
        sid = self.session()
        result = self.send(sid, "fail")
        self.assertEqual(result.status_code, 502)
        self.assertNotIn("private provider detail", result.text)
        self.send(sid, "retry")
        self.assertEqual(self.calls[-1][1], [])


if __name__ == "__main__":
    unittest.main()
