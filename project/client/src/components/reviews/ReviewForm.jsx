// client/src/components/reviews/ReviewForm.jsx
import React, { useState } from 'react';
import { Box, Button, TextField, Paper, Typography } from '@mui/material';

const ReviewForm = ({ onSubmit }) => {
  const [reviewText, setReviewText] = useState('');
  const [rating, setRating] = useState(5);
  const [error, setError] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!reviewText.trim()) {
      setError('Please enter a review.');
      return;
    }
    if (rating < 1 || rating > 5) {
      setError('Rating must be between 1 and 5.');
      return;
    }
    // Call the onSubmit callback with the review data
    onSubmit({ review_text: reviewText, rating });
    // Clear the form fields after successful submission
    setReviewText('');
    setRating(5);
    setError('');
  };

  return (
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
        {error && (
          <Typography variant="body2" color="error" sx={{ mb: 1 }}>
            {error}
          </Typography>
        )}
        <Button type="submit" variant="contained" sx={{ textTransform: 'uppercase' }}>
          Submit Review
        </Button>
      </Box>
    </Paper>
  );
};

export default ReviewForm;
