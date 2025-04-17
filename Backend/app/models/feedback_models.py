# Backend/app/models/feedback_models.py
# ---------------------------
# Comment: Allows users to leave feedback on any component result.
#         Feedback is stored and later used for LLM summarization/tutorials.

from Backend.app import db
from datetime import datetime

class Comment(db.Model):
    __tablename__ = 'comments'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    model_run_id = db.Column(db.Integer, db.ForeignKey('model_run.id'), nullable=True)
    component_type = db.Column(db.String(80), nullable=False)
    component_instance_id = db.Column(db.Integer, nullable=False)
    comment_text = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    model_run = db.relationship('ModelRun', backref=db.backref('comments', lazy=True))

    def __repr__(self):
        return f'<Comment {self.id} on {self.component_type}:{self.component_instance_id}>'
