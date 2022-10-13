import os

import cloudstorage
from google.appengine.api import app_identity


def get(self):
    bucket_name = os.environ.get(
        'BUCKET_NAME', app_identity.get_default_gcs_bucket_name())

    self.response.headers['Content-Type'] = 'text/plain'
    self.response.write(
        'Demo GCS Application running from Version: {}\n'.format(
            os.environ['CURRENT_VERSION_ID']))
    self.response.write('Using bucket name: {}\n\n'.format(bucket_name))


def create_file(self, filename):
    """Create a file."""

    self.response.write('Creating file {}\n'.format(filename))

    # The retry_params specified in the open call will override the default
    # retry params for this particular file handle.
    write_retry_params = cloudstorage.RetryParams(backoff_factor=1.1)
    with cloudstorage.open(
            filename, 'w', content_type='text/plain', options={
                'x-goog-meta-foo': 'foo', 'x-goog-meta-bar': 'bar'},
            retry_params=write_retry_params) as cloudstorage_file:
        cloudstorage_file.write('abcde\n')
        cloudstorage_file.write('f' * 1024 * 4 + '\n')
    self.tmp_filenames_to_clean_up.append(filename)


def read_file(self, filename):
    self.response.write(
        'Abbreviated file content (first line and last 1K):\n')

    with cloudstorage.open(filename) as cloudstorage_file:
        self.response.write(cloudstorage_file.readline())
        cloudstorage_file.seek(-1024, os.SEEK_END)
        self.response.write(cloudstorage_file.read())


def list_bucket(self, bucket):
    """Create several files and paginate through them."""

    self.response.write('Listbucket result:\n')

    # Production apps should set page_size to a practical value.
    page_size = 1
    stats = cloudstorage.listbucket(bucket + '/foo', max_keys=page_size)
    while True:
        count = 0
        for stat in stats:
            count += 1
            self.response.write(repr(stat))
            self.response.write('\n')

        if count != page_size or count == 0:
            break
        stats = cloudstorage.listbucket(
            bucket + '/foo', max_keys=page_size, marker=stat.filename)


def delete_files(self):
    self.response.write('Deleting files...\n')
    for filename in self.tmp_filenames_to_clean_up:
        self.response.write('Deleting file {}\n'.format(filename))
        try:
            cloudstorage.delete(filename)
        except cloudstorage.NotFoundError:
            pass
