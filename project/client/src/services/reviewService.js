// client/src/services/reviewService.js
import api from './api';

/**
 * Retrieves reviews for a specific wine.
 * @param {string} wine_id - The ID of the wine.
 * @returns {Promise<Array>} - An array of review objects.
 */
const getReviews = async (wine_id) => {
  try {
    const response = await api.get('/reviews', { params: { wine_id } });
    return response.data;
  } catch (error) {
    throw error;
  }
};

/**
 * Submits a review for a wine.
 * @param {Object} reviewData - An object containing wine_id, rating, review_text, etc.
 * @param {string} token - The JWT access token for authentication.
 * @returns {Promise<Object>} - The newly created review.
 */
const submitReview = async (reviewData, token) => {
  try {
    const response = await api.post('/reviews', reviewData, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    return response.data;
  } catch (error) {
    throw error;
  }
};

export { getReviews, submitReview };
