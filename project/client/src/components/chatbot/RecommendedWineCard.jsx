// client/src/components/chatbot/RecommendedWineCard.jsx
import React from 'react';
import { Card, CardMedia, CardContent, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const RecommendedWineCard = ({ wine }) => {
  const navigate = useNavigate();

  const handleClick = () => {
    // Navigate using wine.id (ensure your wine object has the id field)
    navigate(`/wine-details/${wine.id}`);
  };

  return (
    <Card
      sx={{
        cursor: 'pointer',
        maxWidth: 300,
        m: 1,
        bgcolor: 'background.paper',
        color: 'text.primary',
        boxShadow: 2,
      }}
      onClick={handleClick}
    >
      <CardMedia
        component="img"
        height="140"
        image={wine.image || 'https://source.unsplash.com/random/300x200/?wine'}
        alt={wine.name}
        sx={{ objectFit: 'cover' }}
      />
      <CardContent>
        <Typography variant="h6" noWrap sx={{ textTransform: 'uppercase', color: 'text.primary' }}>
          {wine.name}
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {wine.country} - {wine.region}
        </Typography>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          Price: ${wine.price}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default RecommendedWineCard;
