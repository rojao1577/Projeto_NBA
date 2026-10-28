from flask import Flask, render_template


def create_app() -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("em_construcao.html", title="Início")

    @app.route("/cv")
    def cv():
        return render_template("em_construcao.html", title="Coeficiente de variação")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
