// client/src/pages/Wines.jsx
import React, { useEffect, useState, useCallback } from 'react';
import {
  Container,
  Typography,
  Card,
  CardMedia,
  CardContent,
  Box
} from '@mui/material';
import Pagination from '@mui/material/Pagination';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { getWines, getWinesCount } from '../services/wineService';

const Wines = () => {
  const [wines, setWines] = useState([]);
  const [limit] = useState(20);    // Fixed number per page
  const [page, setPage] = useState(1);
  const [totalCount, setTotalCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  // Fetch wines for current page
  const fetchWines = useCallback(() => {
    setLoading(true);
    getWines({ limit, page })
      .then((data) => {
        setWines(Array.isArray(data) ? data : []);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching wines:', err);
        setWines([]);
        setLoading(false);
      });
  }, [limit, page]);

  // Fetch total count for pagination
  const fetchWinesCount = useCallback(() => {
    getWinesCount()
      .then((data) => setTotalCount(data.count))
      .catch((err) => {
        console.error('Error fetching wines count:', err);
        setTotalCount(0);
      });
  }, []);

  useEffect(() => {
    fetchWines();
  }, [fetchWines]);

  useEffect(() => {
    fetchWinesCount();
  }, [fetchWinesCount]);

  const handleCardClick = (id) => {
    navigate(`/wine-details/${id}`);
  };

  const handlePageChange = (event, value) => {
    setPage(value);
  };

  const totalPages = Math.ceil(totalCount / limit);

  return (
    <Container
      sx={{
        py: 5,
        position: 'relative',
        zIndex: 2,
        mt: 4,
        // Removed background, borderRadius, and boxShadow for a seamless dark theme
      }}
    >
      <Typography
        variant="h4"
        align="center"
        gutterBottom
        sx={{ textTransform: 'uppercase', color: 'text.primary' }}
      >
        Explore Wines
      </Typography>

      {loading ? (
        <Typography variant="h6" align="center" sx={{ color: 'text.primary' }}>
          Loading...
        </Typography>
      ) : wines.length === 0 ? (
        <Typography variant="h6" align="center" sx={{ color: 'text.primary' }}>
          No wines found. Please check your backend or database.
        </Typography>
      ) : (
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: 2,
          }}
        >
          {wines.map((wine) => (
            <motion.div
              key={wine.id}
              whileHover={{ scale: 1.05 }}
              transition={{ duration: 0.3 }}
            >
              <Card
                sx={{ cursor: 'pointer' }}
                onClick={() => handleCardClick(wine.id)}
              >
                <CardMedia
                  component="img"
                  sx={{
                    height: 300,
                    objectFit: 'contain',
                  }}
                  image={
                    wine.image_url ||
                    'https://source.unsplash.com/random/400x300/?wine'
                  }
                  alt={wine.wine_name}
                />
                <CardContent>
                  <Typography
                    variant="h6"
                    noWrap
                    sx={{ textTransform: 'uppercase', color: 'text.primary' }}
                  >
                    {wine.wine_name}
                  </Typography>
                  <Typography
                    variant="body2"
                    noWrap
                    sx={{ textTransform: 'uppercase', color: 'text.secondary' }}
                  >
                    ${wine.price} • {wine.rating} Stars
                  </Typography>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </Box>
      )}

      {!loading && totalPages > 1 && (
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Pagination
            count={totalPages}
            page={page}
            onChange={handlePageChange}
            color="primary"
          />
        </Box>
      )}
    </Container>
  );
};

export default Wines;
