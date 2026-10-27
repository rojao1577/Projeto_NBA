from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return "Olá! Projeto 1 — Laboratório de Ciência de Dados (NBA 2025-26)."

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
