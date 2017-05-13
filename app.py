from flask import (Flask, g, render_template, flash, redirect, url_for, abort)
from flask_bcrypt import check_password_hash
from flask_login import (LoginManager, login_user, logout_user,
                         login_required, current_user)
from threading import Thread
from werkzeug.utils import secure_filename
from shutil import copy, move, rmtree
from image_processing import ImgProcessor
from sklearn.exceptions import NotFittedError
import os
import forms
import models

DEBUG = True
PORT = 8000
HOST = '0.0.0.0'

app = Flask(__name__)
app.secret_key = 'asdETWESDFDFAsadf'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

IMAGE_PROCESSOR = None
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(CURRENT_PATH, 'images/train/')
TEST_PATH = os.path.join(CURRENT_PATH, 'images/test/')


def async(f):
    def wrapper(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return wrapper


@async
def train_image_processor():
    print("executing task")
    # TODO : Improvement - very ugly way
    global IMAGE_PROCESSOR
    IMAGE_PROCESSOR = ImgProcessor(20)
    IMAGE_PROCESSOR.train_model()


@login_manager.user_loader
def load_user(userid):
    try:
        return models.User.get(models.User.id == userid)
    except models.DoesNotExist:
        return None


@app.before_request
def before_request():
    g.db = models.DATABASE
    g.db.connect()
    g.user = current_user


@app.after_request
def after_request(response):
    g.db.close()
    return response


@app.route('/register', methods=('GET', 'POST'))
def register():
    form = forms.RegisterForm()
    if form.validate_on_submit():
        flash("Registered successfully.", "success")
        models.User.create_user(
            username=form.username.data,
            email=form.email.data,
            password=form.password.data
        )
        train_image_processor()
        return redirect(url_for('index'))
    return render_template('register.html', form=form)


@app.route('/login', methods=('GET', 'POST'))
def login():
    form = forms.LoginForm()
    if form.validate_on_submit():
        try:
            user = models.User.get(models.User.email == form.email.data)
        except models.DoesNotExist:
            flash("Wrong email or password", "error")
        else:
            if check_password_hash(user.password, form.password.data):
                login_user(user)
                flash("Logged in", "success")
                train_image_processor()
                return redirect(url_for('index'))
            else:
                flash("Your email or password does not match", "error")
    return render_template('login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("You have been logged out.")
    return redirect(url_for('login'))


@app.route('/new_post', methods=('GET', 'POST'))
@login_required
def post():
    form = forms.PostForm()
    if form.validate_on_submit():
        f = form.image.data
        filename = secure_filename(f.filename)
        file_path = os.path.join(
            app.instance_path, '..', 'static/img', filename
        )
        f.save(file_path)
        file_desc = form.desc.data.strip()
        post_obj = models.Post.create(user=g.user.id,
                           content=form.content.data.strip(),
                           desc=file_desc,
                           filename=filename)

        flash("Posted!", "success")

        # Prepare test path
        # Ideally requires static files for display
        # to be moved to a testing folder for model
        test_directory = os.path.join(TEST_PATH, file_desc)
        if not os.path.exists(test_directory): os.makedirs(test_directory)
        test_path = os.path.join(test_directory, filename)

        # Copy file from static to test folder
        copy(file_path, test_path)

        # Predict description of image from model
        try:
            prediction = IMAGE_PROCESSOR.test_model()
            predicted_desc = prediction[file_desc]
            query = post_obj.update(predicted_desc=predicted_desc).\
                where(models.Post.filename == filename)
            query.execute()
        except NotFittedError:
            flash("Model not trained yet. Training in background. "
                  "Please wait and try again.")

        # Add the newly tested file to training set for future processing.
        directory = os.path.join(TRAIN_PATH, file_desc)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            # TODO: Improvement
            # If file by the same name exists, don't add it again
            # This could be a bug later, but don't want to break
            # things with a small set.
            move(test_path, os.path.join(TRAIN_PATH, file_desc))
        except Exception as e:
            for i in os.listdir(test_directory):
                os.unlink(os.path.join(test_directory,i))
        os.rmdir(test_directory)
        train_image_processor()

        return redirect(url_for('index'))
    return render_template('create_post.html', form=form)


@app.route('/')
def index():
    stream = models.Post.select().limit(100)
    return render_template('stream.html', stream=stream)


@app.route('/stream')
@app.route('/stream/<username>')
def stream(username=None):
    template = 'stream.html'
    user = None
    stream = None
    if username and username != current_user.username:
        try:
            user = models.User.select().where(models.User.username ** username).get()
        except models.DoesNotExist:
            abort(404)
        else:
            stream = user.posts.limit(100)
    else:
        stream = current_user.get_stream().limit(100)
        user = current_user
    if username:
        template = 'user_stream.html'
    return render_template(template, stream=stream, user=user)


@app.route('/post/<int:post_id>')
def view_post(post_id):
    posts = models.Post.select().where(models.Post.id == post_id)
    if posts.count() == 0:
        abort(404)
    return render_template('post.html', stream=posts)


@app.route('/follow/<username>')
@login_required
def follow(username):
    try:
        to_user = models.User.get(models.User.username ** username)
    except models.DoesNotExist:
        abort(404)
    else:
        try:
            models.Relationship.create(
                from_user=g.user._get_current_object(),
                to_user=to_user
            )
        except models.IntegrityError:
            pass
        else:
            flash("You are now following {}".format(to_user.username), "success")
    return redirect(url_for('stream', username=to_user.username))


@app.route('/unfollow/<username>')
@login_required
def unfollow(username):
    try:
        to_user = models.User.get(models.User.username ** username)
    except models.DoesNotExist:
        abort(404)
    else:
        try:
            models.Relationship.get(
                from_user=g.user._get_current_object(),
                to_user=to_user
            ).delete_instance()
        except models.IntegrityError:
            pass
        else:
            flash("You have unfollowed {}".format(to_user.username), "success")
    return redirect(url_for('stream', username=to_user.username))


@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404


if __name__ == '__main__':
    models.initialize()
    try:
        models.User.create_user(
            username='aditibhatnagar',
            email='aditi@example.com',
            password='password',
            admin=True
        )
    except ValueError:
        pass
    app.run(debug=DEBUG, host=HOST, port=PORT)
