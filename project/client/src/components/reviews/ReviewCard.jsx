// client/src/components/reviews/ReviewCard.jsx
import React from 'react';
import { Paper, Typography, Box } from '@mui/material';

const ReviewCard = ({ review }) => {
  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
        <Typography variant="body2" sx={{ textTransform: 'uppercase' }}>
          Rating: {review.rating} Stars
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {new Date(review.created_at).toLocaleString()}
        </Typography>
      </Box>
      <Typography variant="body1">
        {review.review_text}
      </Typography>
    </Paper>
  );
};

export default ReviewCard;