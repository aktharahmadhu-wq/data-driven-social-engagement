CREATE TABLE content (
 content_id TEXT PRIMARY KEY, date DATE, platform TEXT, content_type TEXT, topic TEXT,
 caption TEXT, hashtags TEXT, day_of_week TEXT, hour_of_day INTEGER, video_length_sec INTEGER,
 follower_count INTEGER, reach INTEGER, views INTEGER, likes INTEGER, comments INTEGER, shares INTEGER,
 saves INTEGER, retention_rate REAL, followers_gained INTEGER, engagement_rate REAL,
 viral_score REAL, viral_class TEXT, format TEXT, hook_type TEXT, posting_time TEXT
);
CREATE TABLE comments (
 comment_id TEXT PRIMARY KEY, content_id TEXT, date DATE, comment_text TEXT,
 sentiment_label TEXT, sentiment_score REAL, relatability_label TEXT, topic TEXT, platform TEXT
);
