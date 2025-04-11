// client/src/components/chatbot/RecommendedWineCard.jsx
import React from 'react';
import { Card, CardMedia, CardContent, Typography, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const RecommendedWineCard = ({ wine }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate(wine.link);
  };

  return (
    <Card sx={{ cursor: 'pointer', maxWidth: 300, m: 1 }} onClick={handleClick}>
      <CardMedia
        component="img"
        height="140"
        image={wine.image || 'https://source.unsplash.com/random/300x200/?wine'}
        alt={wine.name}
      />
      <CardContent>
        <Typography variant="h6" noWrap>
          {wine.name}
        </Typography>
        <Typography variant="body2">
          {wine.country} - {wine.region}
        </Typography>
        <Typography variant="body2">
          Price: ${wine.price}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default RecommendedWineCard;
