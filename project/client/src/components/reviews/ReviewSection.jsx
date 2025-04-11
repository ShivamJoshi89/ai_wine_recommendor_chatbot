import React, { useEffect, useState } from 'react';
import { Box, Typography, Paper, TextField, Button } from '@mui/material';
import { getReviews, submitReview } from '../../services/reviewService';

const ReviewSection = ({ wineId, token }) => {
  const [reviews, setReviews] = useState([]);
  const [reviewText, setReviewText] = useState('');
  const [rating, setRating] = useState(5);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Fetch reviews for this wine
  useEffect(() => {
    setLoading(true);
    getReviews(wineId)
      .then((data) => {
        setReviews(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching reviews:', err);
        setError('Error fetching reviews');
        setLoading(false);
      });
  }, [wineId]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const reviewData = {
        wine_id: wineId,
        rating,
        review_text: reviewText,
      };
      const newReview = await submitReview(reviewData, token);
      // Append the new review to the list
      setReviews((prev) => [...prev, newReview]);
      setReviewText('');
    } catch (err) {
      console.error('Error submitting review:', err);
      setError('Failed to submit review');
    }
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h5" sx={{ textTransform: 'uppercase', mb: 2 }}>
        Reviews
      </Typography>

      {loading ? (
        <Typography variant="body1">Loading reviews...</Typography>
      ) : error ? (
        <Typography variant="body1" color="error">
          {error}
        </Typography>
      ) : (
        reviews.map((review) => (
          <Paper key={review.id} sx={{ p: 2, mb: 2 }}>
            <Typography variant="body2" sx={{ textTransform: 'uppercase' }}>
              Rating: {review.rating} Stars
            </Typography>
            <Typography variant="body1">{review.review_text}</Typography>
            <Typography variant="caption" color="text.secondary">
              {new Date(review.created_at).toLocaleString()}
            </Typography>
          </Paper>
        ))
      )}

      <Paper sx={{ p: 2, mt: 2 }}>
        <Typography variant="h6" sx={{ textTransform: 'uppercase', mb: 2 }}>
          Add a Review
        </Typography>
        <Box component="form" onSubmit={handleSubmit}>
          <TextField
            label="Review"
            variant="outlined"
            fullWidth
            multiline
            rows={3}
            value={reviewText}
            onChange={(e) => setReviewText(e.target.value)}
            sx={{ mb: 2 }}
          />
          <TextField
            label="Rating"
            variant="outlined"
            type="number"
            fullWidth
            value={rating}
            onChange={(e) => setRating(Number(e.target.value))}
            sx={{ mb: 2 }}
            inputProps={{ min: 1, max: 5 }}
          />
          <Button type="submit" variant="contained" sx={{ textTransform: 'uppercase' }}>
            Submit Review
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default ReviewSection;
