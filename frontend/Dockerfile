# Use Nginx as a base image
FROM nginx:alpine

# Remove the default config file
RUN rm /etc/nginx/conf.d/default.conf

# Copy new Nginx config
COPY default.conf /etc/nginx/conf.d/

# Copy the HTML files
COPY index.html /usr/share/nginx/html/

# Expose the default HTTP port
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
